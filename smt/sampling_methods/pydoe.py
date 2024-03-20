"""
Author: Antoine Averland <antoine.averland@onera.fr> and Rémi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.

pyDOE3 sampling methods
"""

import numpy as np
from pyDOE3 import doe_box_behnken, doe_factorial, doe_gsd, doe_plackett_burman

from smt.sampling_methods.sampling_method import SamplingMethod


class PyDoeSamplingMethod(SamplingMethod):
    """
    Base class adapting pyDOE3 designs to SMT SamplingMethod interface
    See https://pydoe3.readthedocs.io/
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nx = self.options["xlimits"].shape[0]
        self.levels = None

    def _compute(self, nt: int = None):
        """
        Use pydoe3 design to produce [nsamples, nx] matrix
        where nsamples depends on the pyDOE3 method and nx is the dimension of x.
        Warning: In pyDOE3 design setting user requested number of points nt is not used
        """
        xlimits = self.options["xlimits"]
        levels = self.levels

        # Retrieve indices from pyDOE3 design
        doe = np.array(self._compute_doe(), dtype=int)

        # Compute scaled values for each x components
        values = np.zeros((self.nx, max(levels)))
        for i in range(self.nx):
            values[i, 0 : levels[i]] = np.linspace(
                xlimits[i, 0],
                xlimits[i, 1],
                num=levels[i],
            )

        # Use indices to shape the result array and fill it with values
        res = np.zeros(doe.shape)
        i = 0
        for idx in doe:
            for j in range(self.nx):
                res[i, j] = values[j, idx[j]]
            i = i + 1

        return res

    def _compute_doe():
        """Returns a matrix (nsamples, nx) of indices.

        Each indices takes a value in [0, nlevels_i-1] where nlevels_i is
        the number of levels of the ith component of x.
        This method has to be overriden by subclasses"""
        raise NotImplementedError(
            "You have to implement DOE computation method _compute_doe()"
        )


class BoxBehnken(PyDoeSamplingMethod):
    """See https://pydoe3.readthedocs.io/en/latest/rsm.html#box-behnken"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Box Behnken design has 3 levels [-1, 0, 1]
        self.levels = [3] * self.nx  # for

    def _compute_doe(self):
        # Increment Box Behnken levels to get indices [0, 1, 2]
        return doe_box_behnken.bbdesign(self.nx) + 1


class Gsd(PyDoeSamplingMethod):
    """See https://pydoe3.readthedocs.io/en/latest/rsm.html#gsd"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.levels = self.options["levels"]

    def _initialize(self, **kwargs):
        self.options.declare(
            "levels",
            types=list,
            desc="number of factor levels per factor in design",
        )
        self.options.declare(
            "reduction",
            types=int,
            default=2,
            desc="Reduction factor (bigger than 1). Larger `reduction` means fewer experiments \
                  in the design and more possible complementary designs",
        )

    def _compute_doe(self):
        levels = self.options["levels"]
        reduction = self.options["reduction"]

        return doe_gsd.gsd(levels, reduction)


class Factorial(PyDoeSamplingMethod):
    """See https://pydoe3.readthedocs.io/en/latest/factorial.html#general-full-factorial"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.levels = self.options["levels"]

    def _initialize(self, **kwargs):
        self.options.declare(
            "levels",
            types=list,
            desc="number of factor levels per factor in design",
        )

    def _compute_doe(self):
        levels = self.options["levels"]
        return doe_factorial.fullfact(levels)


class PlackettBurman(PyDoeSamplingMethod):
    """See https://pydoe3.readthedocs.io/en/latest/factorial.html#plackett-burman"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Plackett Burman design has 2 levels [-1, 1]
        self.levels = [2] * self.nx

    def _compute_doe(self):
        doe = doe_plackett_burman.pbdesign(self.nx)
        # Change -1 level to get indices [0, 1]
        doe[doe < 0] = 0

        return doe
