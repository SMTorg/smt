"""
box_behnken sampling; uses the pyDOE3 package.
"""
from pyDOE3 import doe_box_behnken
from pyDOE3 import doe_gsd
from pyDOE3 import doe_factorial
from pyDOE3 import doe_plackett_burman
import numpy as np

from smt.sampling_methods.sampling_method import SamplingMethod


class PyDoeSamplingMethod(SamplingMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nx = self.options["xlimits"].shape[0]
        self.levels = None

    def _compute(self, nt: int = None):
        xlimits = self.options["xlimits"]
        levels = self.levels

        doe = self._compute_doe()
        indices = np.array(doe, dtype=int)
        print(indices)

        values = np.zeros((self.nx, max(levels)))
        for i in range(self.nx):
            values[i, 0 : levels[i]] = np.linspace(
                xlimits[i, 0], xlimits[i, 1], num=levels[i]
            )
        print(values)

        res = np.zeros(doe.shape)
        i = 0
        for idx in indices:
            for j in range(self.nx):
                res[i, j] = values[j, idx[j]]
            i = i + 1

        return res

    def _compute_doe():
        raise NotImplementedError(
            "You have to implement DOE generation method _compute_doe()"
        )


class BoxBehnken(PyDoeSamplingMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.levels = [3] * self.nx  # for box behnken the number of levels is fixed

    def _compute_doe(self):
        box_behnken_doe = (
            doe_box_behnken.bbdesign(self.nx) + 1
        )  # We have to increment the elements of doe_box_behnken to have the indices

        return box_behnken_doe


class Gsd(PyDoeSamplingMethod):
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
            desc="Reduction factor (bigger than 1). Larger `reduction` means fewer experiments in the design and more possible complementary designs",
        )

    def _compute_doe(self):
        levels = self.options["levels"]
        reduction = self.options["reduction"]
        gsd_doe = doe_gsd.gsd(levels, reduction)

        return gsd_doe


class Factorial(PyDoeSamplingMethod):
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
        factorial_doe = doe_factorial.fullfact(levels)

        return factorial_doe


class PlackettBurman(PyDoeSamplingMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.levels = [2] * self.nx  # for plackett burman the number of levels is fixed

    def _compute_doe(self):
        plackett_burman_doe = doe_plackett_burman.pbdesign(self.nx)
        ny = plackett_burman_doe.shape[1]
        nb_rows = 4 * (
            int(self.nx / 4) + 1
        )  # calculate the correct number of rows (multiple of 4)
        for i in range(nb_rows):
            for j in range(ny):
                if plackett_burman_doe[i, j] == -1:
                    plackett_burman_doe[i, j] = 0

        return plackett_burman_doe
