"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
        
This package is distributed under New BSD license.
"""

import numpy as np
import unittest


class SMTestCase(unittest.TestCase):
    def assert_error(self, computed, desired, atol=1e-15, rtol=1e-15):
        """
        Check relative error of a scalar or array.

        Parameters
        ----------
        computed : float or ndarray
            Computed value; should be the same type and shape as desired.
        desired : float or ndarray
            Desired value; should be the same type and shape as computed.
        atol : float
            Acceptable absolute error. Default is 1e-15.
        rtol : float
            Acceptable relative error. Default is 1e-15.
        """
        abs_error = np.linalg.norm(computed - desired)
        if np.linalg.norm(desired) > 0:
            rel_error = abs_error / np.linalg.norm(desired)
        else:
            rel_error = abs_error
        if abs_error > atol and rel_error > rtol:
            self.fail(
                "computed %s, desired %s, abs error %s, rel error %s, atol %s, rtol %s"
                % (
                    np.linalg.norm(computed),
                    np.linalg.norm(desired),
                    abs_error,
                    rel_error,
                    atol,
                    rtol,
                )
            )
