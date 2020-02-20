"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest
import numpy as np
from smt.surrogate_models import KRG
from smt.problems import Sphere
from smt.sampling_methods import LHS


class TestKRG(unittest.TestCase):
    def test_predict_output_shape(self):
        x = np.random.random((10, 3))
        y = np.random.random((10, 2))

        kriging = KRG()
        kriging.set_training_values(x, y)
        kriging.train()

        val = kriging.predict_values(x)
        self.assertEqual(y.shape, val.shape)

        var = kriging.predict_variances(x)
        self.assertEqual(y.shape, var.shape)

    def test_derivatives(self):
        # Construction of the DOE
        fun = Sphere(ndim=2)
        sampling = LHS(xlimits=fun.xlimits, criterion="m")
        xt = sampling(20)
        yt = fun(xt)

        # Compute the training derivatives
        for i in range(2):
            yd = fun(xt, kx=i)
            yt = np.concatenate((yt, yd), axis=1)

        # check KRG models
        sm_krg_c = KRG(poly="constant", print_global=False)
        sm_krg_c.set_training_values(xt, yt[:, 0])
        sm_krg_c.train()
        TestKRG._check_derivatives(sm_krg_c, xt, yt)

        sm_krg_l = KRG(poly="linear", print_global=False)
        sm_krg_l.set_training_values(xt, yt[:, 0])
        sm_krg_l.train()
        TestKRG._check_derivatives(sm_krg_l, xt, yt)

    @staticmethod
    def _check_derivatives(sm, xt, yt, i=10):
        # Compares three derivatives at i-th traning point
        # 1. Training derivative: "exact" value
        # 2. Predicted derivative: obtaied by sm.predict_derivatives()

        # testing point
        x_test = xt[i].reshape((1, 2))

        # 2. derivatives prediction by surrogate
        dydx_predict = np.zeros(2)
        dydx_predict[0] = sm.predict_derivatives(x_test, kx=0)[0]
        dydx_predict[1] = sm.predict_derivatives(x_test, kx=1)[0]
        print(dydx_predict)
        print(yt[i, 1:])

        # compare results
        np.testing.assert_allclose(yt[i, 1:], dydx_predict, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
