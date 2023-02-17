"""
Author: Paul Saves

"""

import unittest
import numpy as np
from smt.utils.kriging import XSpecs
from smt.utils.mixed_integer import (
    ORD_TYPE,
    FLOAT_TYPE,
    ENUM_TYPE,
    unfold_xlimits_with_continuous_limits,
)
from smt.utils.kriging import (
    NEUTRAL_ROLE,
    DECREED_ROLE,
    META_ROLE,
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
        self.assertEqual([FLOAT_TYPE], xspecs.types)

    def test_xspecs_check_consistency(self):
        xtypes = [FLOAT_TYPE, (ENUM_TYPE, 3)]
        xlimits = [[-10, 10], ["blue", "red", "green"]]
        xroles = [[DECREED_ROLE, META_ROLE, DECREED_ROLE]]  # Bad dimension
        with self.assertRaises(ValueError):
            XSpecs(xtypes=xtypes, xlimits=xlimits, xroles=xroles)

        xtypes = [FLOAT_TYPE, (ENUM_TYPE, 3), ORD_TYPE]
        xlimits = [[-10, 10], ["blue", "red"], [-10, 10]]  # Bad enum
        with self.assertRaises(ValueError):
            XSpecs(xtypes=xtypes, xlimits=xlimits)

        xtypes = [FLOAT_TYPE, (ENUM_TYPE, 2), (ENUM_TYPE, 3), ORD_TYPE]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], ["0", "4", "3"]],
            dtype="object",
        )
        with self.assertRaises(ValueError):
            XSpecs(xtypes=xtypes, xlimits=xlimits)


if __name__ == "__main__":
    unittest.main()
