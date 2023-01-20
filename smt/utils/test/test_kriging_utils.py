"""
Author: Paul Saves

"""

import unittest
import numpy as np
from smt.utils.kriging_utils import XSpecs
from smt.utils.mixed_integer import (
    ORD,
    FLOAT,
    ENUM,
    unfold_xlimits_with_continuous_limits,
)


class Test(unittest.TestCase):
    def test_x_specs(self):
        # xlimits has to be specified
        with self.assertRaises(ValueError):
            XSpecs(xtypes=[3])

        # check consistency: badly-formed xlimits
        with self.assertRaises(TypeError):
            XSpecs(xlimits=[3])

        # ok default to float
        xspecs = XSpecs(xlimits=[[0, 1]])
        self.assertEqual([FLOAT], xspecs.types)

    def test_xspecs_check_consistency(self):
        xtypes = [FLOAT, (ENUM, 3), ORD]
        xlimits = [[-10, 10], ["blue", "red", "green"]]  # Bad dimension
        with self.assertRaises(ValueError):
            XSpecs(xtypes=xtypes, xlimits=xlimits)

        xtypes = [FLOAT, (ENUM, 3), ORD]
        xlimits = [[-10, 10], ["blue", "red"], [-10, 10]]  # Bad enum
        with self.assertRaises(ValueError):
            XSpecs(xtypes=xtypes, xlimits=xlimits)

        xtypes = [FLOAT, (ENUM, 2), (ENUM, 3), ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], ["0", "4", "3"]],
            dtype="object",
        )
        with self.assertRaises(ValueError):
            XSpecs(xtypes=xtypes, xlimits=xlimits)


if __name__ == "__main__":
    unittest.main()
