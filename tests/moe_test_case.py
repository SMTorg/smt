"""
This files contains functions which are used to test the mixture of experts
"""
import unittest
import numpy as np


class MOETestCase(unittest.TestCase):
    """
    Class which contains functions which are used to test the code of MoE.
    list of functions :
    - assert_error
    """

    def assert_error(self, computed, desired, error_margin=0):
        """
        Check relative error of a scalar.
        Parameters
        ----------
        computed : float
            Computed value.
        desired : float
            Desired value.
        error_margin : float
            Acceptable absolute error. Default is 0.
            If 0, computed value should be inferior to desired value
        """
        if error_margin > 0:
            abs_error = np.linalg.norm(computed - desired)
        else:
            abs_error = computed - desired
        if abs_error > error_margin:
            self.fail('computed %s, desired %s, abs_error %s, error_margin %s'
                      % (computed, desired, abs_error, error_margin))
