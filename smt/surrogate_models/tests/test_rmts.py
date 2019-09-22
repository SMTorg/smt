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
    def test_linear_search(self):
        # xt = np.random.rand(5, 1) * 10.0
        xt = np.array([[3.6566495, 4.64266046, 7.23645433, 6.04862594, 8.85571712]]).T
        yt = function_test_1d(xt)

        xlimits = np.array([[0.0, 25.0]])

        smref = RMTB(xlimits=xlimits, print_global=False)
        smref.set_training_values(xt, yt)
        with Silence():
            smref.train()

        xref = np.array([[0.0, 6.25, 12.5, 18.75, 25.0]]).T
        yref = smref.predict_values(xref)

        sms = {}
        for ls in ["bracketed", "cubic", "quadratic", "null"]:
            sms[ls] = RMTB(xlimits=xlimits, line_search=ls, print_global=False)
            sms[ls].set_training_values(xt, yt)

            with Silence():
                sms[ls].train()

            error = compute_rms_error(sms[ls], xref, yref)
            self.assert_error(error, 0.0, 1e-1)

    def test_linear_solver(self):
        xt = np.array([[3.6566495, 4.64266046, 7.23645433, 6.04862594, 8.85571712]]).T
        yt = function_test_1d(xt)

        xlimits = np.array([[0.0, 25.0]])

        smref = RMTB(xlimits=xlimits, print_global=False)
        smref.set_training_values(xt, yt)
        with Silence():
            smref.train()

        xref = np.array([[0.0, 6.25, 12.5, 18.75, 25.0]]).T
        yref = smref.predict_values(xref)

        sms = {}
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
            sms[ls] = RMTB(xlimits=xlimits, solver=ls, print_global=False)
            sms[ls].set_training_values(xt, yt)

            with Silence():
                sms[ls].train()

            error = compute_rms_error(sms[ls], xref, yref)
            self.assert_error(error, 0.0, 1.1)


if __name__ == "__main__":
    unittest.main()
