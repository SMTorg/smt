"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest
import numpy as np
from smt.surrogate_models import MGP
from smt.sampling_methods import FullFactorial


class TestMGP(unittest.TestCase):
    def test_predict_output_shape(self):
        x = np.random.random((10, 3))
        y = np.random.random((10,))

        kriging = MGP(n_comp=2)
        kriging.set_training_values(x, y)
        kriging.train()

        val = kriging.predict_values(x)
        self.assertEqual(y.shape, val.shape)

        var = kriging.predict_variances(x)
        self.assertEqual(y.shape, var.shape)

    def test_good_subspace(self):
        ndim = 5
        fun = lambda x: np.sum(x, axis=1)
        sampling = FullFactorial(xlimits=np.asarray([(-1, 1)] * ndim))
        xt = sampling(250)
        yt = np.atleast_2d(fun(xt)).T

        sm_krg_c = MGP(poly="constant", print_global=False, n_comp=1)
        sm_krg_c.set_training_values(xt, yt[:, 0])
        sm_krg_c.train()
        np.testing.assert_allclose(
            sm_krg_c.embedding["C"] / np.max(sm_krg_c.embedding["C"]),
            np.atleast_2d([1] * ndim).T,
            rtol=1e-4,
            atol=0,
        )
        np.testing.assert_almost_equal(
            np.linalg.norm(np.linalg.inv(sm_krg_c.inv_sigma_R)), 0, decimal=4
        )


if __name__ == "__main__":
    unittest.main()
