"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest
import numpy as np
from smt.surrogate_models import KRG
from smt.problems import Sphere
from smt.sampling_methods import FullFactorial, LHS


class TestKRG(unittest.TestCase):
    def test_predict_output_shape(self):
        d, n = (3, 10)
        sx = LHS(
            xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), d, axis=0),
            criterion="m",
            random_state=42,
        )
        x = sx(n)
        sy = LHS(
            xlimits=np.repeat(np.atleast_2d([0.0, 1.0]), 2, axis=0),
            criterion="m",
            random_state=42,
        )
        y = sy(n)

        kriging = KRG()
        kriging.set_training_values(x, y)
        kriging.train()

        val = kriging.predict_values(x)
        self.assertEqual(y.shape, val.shape)

        var = kriging.predict_variances(x)
        self.assertEqual(y.shape, var.shape)

    def test_derivatives(self):
        # Construction of the DOE
        ndim = 4
        fun = Sphere(ndim=ndim)
        sampling = FullFactorial(xlimits=fun.xlimits)
        xt = sampling(100)
        yt = fun(xt)

        # Compute the training derivatives
        for i in range(ndim):
            yd = fun(xt, kx=i)
            yt = np.concatenate((yt, yd), axis=1)

        # check KRG models
        sm_krg_c = KRG(poly="constant", print_global=False)
        sm_krg_c.set_training_values(xt, yt[:, 0])
        sm_krg_c.train()
        TestKRG._check_derivatives(sm_krg_c, xt, yt, ndim)

        sm_krg_l = KRG(poly="linear", print_global=False)
        sm_krg_l.set_training_values(xt, yt[:, 0])
        sm_krg_l.train()
        TestKRG._check_derivatives(sm_krg_l, xt, yt, ndim)

    @staticmethod
    def _check_derivatives(sm, xt, yt, ndim, i=10):
        # Compares three derivatives at i-th traning point
        # 1. Training derivative: "exact" value
        # 2. Predicted derivative: obtaied by sm.predict_derivatives()

        # testing point
        x_test = xt[i].reshape((1, ndim))

        # 2. derivatives prediction by surrogate
        dydx_predict = np.zeros(ndim)
        for j in range(ndim):
            dydx_predict[j] = sm.predict_derivatives(x_test, kx=j)[0]
            dydx_predict[j] = sm.predict_derivatives(x_test, kx=j)[0]
            dydx_predict[j] = sm.predict_derivatives(x_test, kx=j)[0]
            dydx_predict[j] = sm.predict_derivatives(x_test, kx=j)[0]
        print(dydx_predict)
        print(yt[i, 1:])

        # compare results
        np.testing.assert_allclose(yt[i, 1:], dydx_predict, atol=2e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
