"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt

from smt.utils.silence import Silence
from smt.utils.sm_test_case import SMTestCase
from smt.utils import compute_rms_error
from smt.surrogate_models import RMTB, RMTC


def function_test_1d(x):
    # function xsinx
    x = np.reshape(x, (-1,))
    y = np.zeros(x.shape)
    y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
    return y.reshape((-1, 1))


class TestRMTS(SMTestCase):
    def setUp(self):
        # xt = np.random.rand(5, 1) * 10.0
        self.xt = np.array(
            [[3.6566495, 4.64266046, 7.23645433, 6.04862594, 8.85571712]]
        ).T
        self.yt = function_test_1d(self.xt)

        self.xlimits = np.array([[0.0, 25.0]])
        smref = RMTB(xlimits=self.xlimits, print_global=False)
        smref.set_training_values(self.xt, self.yt)
        with Silence():
            smref.train()

        self.xref = np.array([[0.0, 6.25, 12.5, 18.75, 25.0]]).T
        self.yref = smref.predict_values(self.xref)

        self.sms = {}

    def test_linear_search(self):
        for ls in ["bracketed", "cubic", "quadratic", "null"]:
            self.sms[ls] = RMTB(
                xlimits=self.xlimits, line_search=ls, print_global=False
            )
            self.sms[ls].set_training_values(self.xt, self.yt)

            with Silence():
                self.sms[ls].train()

            error = compute_rms_error(self.sms[ls], self.xref, self.yref)
            self.assert_error(error, 0.0, 1e-1)

    def test_linear_solver(self):
        for ls in [
            "krylov-dense",
            "dense-chol",
            "lu",
            "ilu",
            "krylov",
            "krylov-lu",
            "krylov-mg",
            "gs",
            "jacobi",
            "mg",
            "null",
        ]:
            self.sms[ls] = RMTB(xlimits=self.xlimits, solver=ls, print_global=False)
            self.sms[ls].set_training_values(self.xt, self.yt)

            with Silence():
                self.sms[ls].train()

            error = compute_rms_error(self.sms[ls], self.xref, self.yref)
            self.assert_error(error, 0.0, 1.1)


if __name__ == "__main__":
    unittest.main()
