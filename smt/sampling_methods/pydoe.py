"""
box_behnken sampling; uses the pyDOE3 package.
"""
from pyDOE3 import doe_box_behnken
from pyDOE3 import doe_gsd
import numpy as np

from smt.sampling_methods.sampling_method import SamplingMethod

class BoxBehnken(SamplingMethod):

    def _compute (self, nt):
        nlevels = 3

        xlimits = self.options["xlimits"]
        nx = xlimits.shape[0]

        box_behnken = doe_box_behnken.bbdesign(nx)
        indices = np.array(box_behnken + 1, dtype=int)
        print(indices)

        values = np.zeros((nx, nlevels))
        for i in range(nx):
            values[i, :] = np.linspace(xlimits[i, 0], xlimits[i, 1], num=nlevels)
        print(values)

        res = np.zeros(box_behnken.shape)  
        i = 0
        for idx in indices:  
            for j in range(nx):
                res[i, j] = values[j, idx[j]]
            i = i+1

        return res


class Gsd(SamplingMethod):

    def _initialize(self, **kwargs):
        self.options.declare(
                "levels",
                types= list,
                desc="number of factor levels per factor in design",
            )
        self.options.declare(
                "reduction",
                types=int,
                desc="Reduction factor (bigger than 1). Larger `reduction` means fewer experiments in the design and more possible complementary designs",
            )

    def _compute (self, nt):
        levels = self.options["levels"]
        reduction = self.options["reduction"]
        xlimits = self.options["xlimits"]
        nx = xlimits.shape[0]

        gsd = doe_gsd.gsd(levels, reduction)
        indices = np.array(gsd, dtype=int)
        print(indices)

        values = np.zeros((nx, max(levels)))
        for i in range(nx):
            values[i, 0:levels[i]] = np.linspace(xlimits[i, 0], xlimits[i, 1], num=levels[i])
        print(values)

        res = np.zeros(gsd.shape)  
        i = 0
        for idx in indices:  
            for j in range(nx):
                res[i, j] = values[j, idx[j]]
            i = i+1

        return res
