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
        inst = XSpecs()
        inst["xtypes"] = [3]
        self.assertEqual(None, inst["xlimits"])
        self.assertEqual([3], inst["xtypes"])
        eq_check = False
        try:
            inst["a"] = [3]
        except AssertionError:
            eq_check = True
        self.assertEqual(True, eq_check)
        newdic = {"xlimits": [4], "xtypes": None}
        inst.update(newdic)
        inst2 = inst.clone()
        self.assertEqual([4], inst2["xlimits"])

    def test_check_xspec_consistency(self):
        xtypes = [FLOAT, (ENUM, 3), ORD]
        xlimits = [[-10, 10], ["blue", "red", "green"]]  # Bad dimension
        xspecs = XSpecs()

        xspecs["xtypes"] = xtypes
        xspecs["xlimits"] = xlimits

        with self.assertRaises(ValueError):
            xspecs.check_xspec_consistency()

        xtypes = [FLOAT, (ENUM, 3), ORD]
        xlimits = [[-10, 10], ["blue", "red"], [-10, 10]]  # Bad enum
        xspecs = XSpecs()

        xspecs["xtypes"] = xtypes
        xspecs["xlimits"] = xlimits
        with self.assertRaises(ValueError):
            xspecs.check_xspec_consistency()

        xtypes = [FLOAT, (ENUM, 2), (ENUM, 3), ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], ["0", "4", "3"]],
            dtype="object",
        )
        xspecs = XSpecs()

        xspecs["xtypes"] = xtypes
        xspecs["xlimits"] = xlimits
        l = unfold_xlimits_with_continuous_limits(xspecs)
        with self.assertRaises(ValueError):
            xspecs.check_xspec_consistency()


if __name__ == "__main__":
    unittest.main()
