"""
Author: Remi Lafage <<remi.lafage@onera.fr>>, Frederic Zahle

This package is distributed under New BSD license.
"""

import unittest
import numpy as np

from smt.surrogate_models import QP, KRG
from smt.examples.rans_crm_wing.rans_crm_wing import (
    get_rans_crm_wing,
    plot_rans_crm_wing,
)


class TestQP(unittest.TestCase):
    def test_ny(self):
        xt, yt, _ = get_rans_crm_wing()

        interp = QP()
        interp.set_training_values(xt, yt)
        interp.train()
        v0 = np.zeros((4, 2))
        for ix, i in enumerate([10, 11, 12, 13]):
            v0[ix, :] = interp.predict_values(np.atleast_2d(xt[i, :]))
        v1 = interp.predict_values(np.atleast_2d(xt[10:14, :]))

        expected_diff = np.zeros((4, 2))
        np.testing.assert_allclose(v1 - v0, expected_diff, atol=1e-15)
