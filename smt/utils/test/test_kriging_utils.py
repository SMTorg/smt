"""
Author: Paul Saves

"""

import unittest
import numpy as np
from smt.utils.kriging_utils import XSpecs


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


if __name__ == "__main__":
    unittest.main()
