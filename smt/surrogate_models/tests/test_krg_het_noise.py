"""
Author: Andres Lopez-Lopera <<andres.lopez_lopera@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest
import numpy as np

from smt.surrogate_models import KRG
from smt.utils.sm_test_case import SMTestCase
from smt.utils import compute_rms_error


class Test(SMTestCase):
    def test_predict_output(self):
        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 1.1, 1.0])

        # Adding noisy repetitions
        np.random.seed(6)
        yt_std_rand = np.std(yt) * np.random.uniform(size=yt.shape)
        xt_full = np.array(3 * xt.tolist())
        yt_full = np.concatenate((yt, yt + 0.2 * yt_std_rand, yt - 0.2 * yt_std_rand))

        sm = KRG(theta0=[1.0], eval_noise=True, use_het_noise=True, n_start=1)
        sm.set_training_values(xt_full, yt_full)
        sm.train()

        yt = yt.reshape(-1, 1)
        y = sm.predict_values(xt)
        t_error = np.linalg.norm(y - yt) / np.linalg.norm(yt)
        self.assert_error(t_error, 0.0, 1e-2)


if __name__ == "__main__":
    unittest.main()
